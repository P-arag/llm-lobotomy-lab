'use client'

import React, { useState, useRef, FC, ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs))
}

const Button = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: string; size?: string }
>(({ className, variant, size, ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "inline-flex items-center justify-between rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      variant === "outline"
        ? "border border-input bg-background hover:bg-accent hover:text-accent-foreground"
        : variant === "ghost"
        ? "hover:bg-accent hover:text-accent-foreground"
        : variant === "link"
        ? "text-primary underline-offset-4 hover:underline"
        : "bg-primary text-primary-foreground hover:bg-primary/90",
      size === "sm" ? "h-9 px-3" : size === "lg" ? "h-11 px-8" : size === "icon" ? "h-10 w-10" : "h-10 px-4 py-2",
      className
    )}
    {...props}
  />
))
Button.displayName = "Button"

function useClickOutside(ref: React.RefObject<HTMLElement | null>, handler: () => void) {
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        handler()
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [ref, handler])
}

export interface ModelItem {
  id: string
  name: string
  tag?: string
}

interface AnimatedModelDropdownProps {
  items?: ModelItem[]
  value?: string
  onChange?: (item: ModelItem) => void
  className?: string
}

const DEFAULT_MODELS: ModelItem[] = [
  { id: 'qwen-3b', name: 'Qwen2.5-3B-abliterated', tag: 'Fast / Local' },
  { id: 'llama-3b', name: 'Llama-3.2-3B-abliterated', tag: 'Balanced' },
  { id: 'qwen-27b', name: 'Qwen-2.5-27B-abliterated', tag: 'Cloud / High-VRAM' },
]

export default function AnimatedModelDropdown({
  items = DEFAULT_MODELS,
  value = 'qwen-3b',
  onChange,
  className,
}: AnimatedModelDropdownProps) {
  const [selectedId, setSelectedId] = useState(value)
  const [isOpen, setIsOpen] = useState(false)

  const selectedItem = items.find((m) => m.id === selectedId) || items[0]

  const handleSelect = (item: ModelItem) => {
    setSelectedId(item.id)
    setIsOpen(false)
    onChange?.(item)
  }

  return (
    <OnClickOutside onClickOutside={() => setIsOpen(false)}>
      <div
        data-state={isOpen ? 'open' : 'closed'}
        className={cn('group relative w-full', className)}
      >
        <Button
          type="button"
          variant="outline"
          aria-haspopup="listbox"
          aria-expanded={isOpen}
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between border-border bg-input px-3 py-2 text-xs font-mono text-foreground hover:border-accent"
        >
          <span className="truncate">{selectedItem.name}</span>
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            <ChevronDown className="h-4 w-4 opacity-60 ml-2" />
          </motion.div>
        </Button>

        <AnimatePresence>
          {isOpen && (
            <motion.div
              role="listbox"
              initial={{ opacity: 0, y: -8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.96 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
              className={cn(
                'absolute top-[calc(100%+0.35rem)] left-0 z-50 w-full',
                'overflow-hidden rounded-md',
                'bg-zinc-900 border border-border',
                'shadow-xl backdrop-blur-sm'
              )}
            >
              <motion.div
                initial="hidden"
                animate="visible"
                variants={{
                  visible: {
                    transition: {
                      staggerChildren: 0.03,
                    },
                  },
                }}
              >
                {items.map((item) => (
                  <motion.button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelect(item)}
                    variants={{
                      hidden: { opacity: 0, x: -10 },
                      visible: { opacity: 1, x: 0 },
                    }}
                    className={cn(
                      'flex items-center justify-between w-full px-3 py-2 text-xs font-mono text-left',
                      'border-b border-border/50 last:border-b-0',
                      'bg-zinc-900 hover:bg-zinc-800/80 transition-colors duration-150',
                      item.id === selectedId ? 'text-accent bg-zinc-800/50 font-semibold' : 'text-zinc-300'
                    )}
                  >
                    <span>{item.name}</span>
                    {item.tag && (
                      <span className="text-[10px] text-zinc-500 font-sans">{item.tag}</span>
                    )}
                  </motion.button>
                ))}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </OnClickOutside>
  )
}

interface Props {
  children: ReactNode
  onClickOutside: () => void
  classes?: string
}

const OnClickOutside: FC<Props> = ({ children, onClickOutside, classes }) => {
  const wrapperRef = useRef<HTMLDivElement>(null)
  useClickOutside(wrapperRef, onClickOutside)

  return (
    <div ref={wrapperRef} className={cn(classes)}>
      {children}
    </div>
  )
}
